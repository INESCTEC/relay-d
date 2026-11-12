import sys
import os

import cv2  # noqa: F401  (imported first so its Qt env side effects can be undone)
from relay_d.utils.qt_env import sanitize_qt_env
sanitize_qt_env()

import rclpy
from rclpy.node import Node

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QFileSystemWatcher

from relay_d.utils import user_settings
from .ctrl_c import close_on_ctrl_c
from .first_run import ask_save_folder
from .settings_apply import live_changes, needs_config, ros_domain_to_set, startup_page
from .ui.settings.config_manager import config
from .utils.custom_message_box import CustomMessageBox

try:
    from relay_d.utils.coloring_logger import logger, apply_user_level
except ImportError:
    # Fallback to standard logging if relay_d isn't installed/importable
    import logging
    logger = logging.getLogger(__name__)

    def apply_user_level():
        pass

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import resources_rc
except ImportError:
    logger.error("Failed to import resources_rc. Make sure the .qrc file is compiled correctly.")
    pass

try:
    from .ui.config.config_page import ConfigPage
    from .ui.record.record_page import RecordPage
    from .ui.postprocess.postprocess_page import PostProcessPage
    from .ui.settings.settings_page import SettingsPage
    from .ui.review.review_page import ReviewPage
    from .ui.train.train_page import TrainPage
    from .ui.run.run_page import RunPage
except ImportError as e:
    logger.error(f"Failed to import one or more UI pages: {e}")
    ConfigPage = RecordPage = PostProcessPage = SettingsPage = ReviewPage = TrainPage = RunPage = None


class UI(QMainWindow):
    def __init__(self, node):
        super(UI, self).__init__()
        self.node = node
        self.ros_nodes = [node]

        # ROS spin timer - this is critical for receiving ROS messages
        self.ros_spin_timer = QTimer()
        self.ros_spin_timer.timeout.connect(self.spin_once)
        self.ros_spin_timer.start(int(user_settings.get("ros_spin_ms")))  # setting: how often ROS callbacks run

        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.on_resize_finished)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file_path = os.path.join(script_dir, "ui/containers/mainWindow.ui")

        uic.loadUi(ui_file_path, self)

        self.setup_menu_buttons([
            "btn_start",
            "btn_data_post_process",
            "btn_import",
            "btn_settings",
            "btn_review",
            "btn_train",
            "btn_run",
        ])

        # After setup_menu_buttons(self, menu_buttons)
        self.btn_help.clicked.connect(
            lambda: self.stackedWidget.setCurrentWidget(self.page_help)
        )

        try:
            if ConfigPage is not None:
                self.page_config = ConfigPage(self)
            if SettingsPage is not None:
                self.page_settings_ui = SettingsPage(self)
            if RecordPage is not None:
                parser = (
                    self.page_config.yaml_parser
                    if hasattr(self, "page_config") and self.page_config is not None
                    else None
                )
                settings = (
                    self.page_settings_ui
                    if hasattr(self, "page_settings_ui")
                    and self.page_settings_ui is not None
                    else None
                )
                self.page_record = RecordPage(
                    self,
                    parser,
                    settings,
                )
            if PostProcessPage is not None:
                parser = (
                    self.page_config.yaml_parser
                    if hasattr(self, "page_config") and self.page_config is not None
                    else None
                )
                settings = (
                    self.page_settings_ui
                    if hasattr(self, "page_settings_ui")
                    and self.page_settings_ui is not None
                    else None
                )
                self.page_postprocess = PostProcessPage(
                    self,
                    parser,
                    settings,
                )
            if ReviewPage is not None:
                settings = (
                    self.page_settings_ui
                    if hasattr(self, "page_settings_ui")
                    and self.page_settings_ui is not None
                    else None
                )
                self.page_review_ui = ReviewPage(
                    self,
                    record_page=getattr(self, "page_record", None),
                    settings_page=settings,
                )
            if TrainPage is not None:
                self.page_train_ui = TrainPage(
                    self, settings_page=getattr(self, "page_settings_ui", None)
                )
            if RunPage is not None:
                self.page_run_ui = RunPage(self)
            if hasattr(self, "page_record") and hasattr(self, "page_config"):
                self.page_record.connect_to_config_signals(
                    self.page_config.signal_manager
                )
            if hasattr(self, "page_postprocess") and hasattr(self, "page_config"):
                self.page_postprocess.connect_to_config_signals(
                    self.page_config.signal_manager
                )
        except Exception as e:
            logger.error(f"Error setting up pages: {e}")
            sys.exit(1)

        theme = user_settings.get("theme")
        self.load_theme(theme)

        # Configuração do Live Reload para QSS
        self.qss_watcher = QFileSystemWatcher()
        self._update_watcher_path(theme)
        self.qss_watcher.fileChanged.connect(self._on_qss_changed)
        self._theme = theme
        config.settings_changed.connect(self._on_settings_changed)

        # Load the settings
        if hasattr(self, "page_settings_ui"):
            self.page_settings_ui.load_settings()

        # Change the current widget to the start page (setting: startup_page)
        startup_blocked = False
        try:
            startup = getattr(self, f"page_{startup_page(user_settings.values())}", self.page_help)
            startup_blocked = self._needs_config(startup)  # starting on the Record page without a config: open Config instead
            self.stackedWidget.setCurrentWidget(self.page_import if startup_blocked else startup)
        except Exception as e:
            logger.error(f"Error setting initial page: {e}")

        self.show()
        if not ask_save_folder(self):  # modal: nothing else can be used until a folder is chosen
            QTimer.singleShot(0, self.close)
        elif startup_blocked:
            self._warn_no_config()

    def register_ros_node(self, node):
        """Register a ROS node to be spun"""
        if node not in self.ros_nodes:
            self.ros_nodes.append(node)
            logger.info(f"Registered ROS node: {node.get_name()}")

    def unregister_ros_node(self, node):
        """Stop spinning a previously-registered ROS node (e.g. after it has
        been destroyed)."""
        if node in self.ros_nodes:
            self.ros_nodes.remove(node)

    def spin_once(self):
        """Spin all ROS nodes to process callbacks"""
        for node in self.ros_nodes:
            try:
                rclpy.spin_once(node, timeout_sec=0)
            except Exception as e:
                logger.debug(f"Error spinning ROS node: {e}")

    def setup_menu_buttons(self, menu_buttons):
        for btn_name in menu_buttons:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.clicked.connect(
                    lambda checked, name=btn_name: self.on_menu_button_clicked(name)
                )
            else:
                logger.warning(f"Menu button not found: {btn_name}")

    def on_menu_button_clicked(self, btn_name):
        logger.info(f"Menu button clicked: {btn_name}")
        page_mapping = {
            "btn_start": self.page_start,
            "btn_data_post_process": self.page_data,
            "btn_import": self.page_import,
            "btn_settings": self.page_settings,
            "btn_help": self.page_help,
            "btn_review": self.page_review,
            "btn_train": self.page_train,
            "btn_run": self.page_run,
        }
        if btn_name in page_mapping:
            try:
                if btn_name == "btn_review" and hasattr(self, "page_review_ui"):
                    self.page_review_ui.sync_with_current_session()
                    self.page_review_ui.refresh_demo_list()
                target = page_mapping[btn_name]
                if self._needs_config(target):  # the Record page does nothing without a config
                    self._warn_no_config()
                    target = self.page_import
                self.stackedWidget.setCurrentWidget(target)
            except Exception as e:
                logger.error(f"Error switching to page for {btn_name}: {e}")
        else:
            logger.warning(f"No page mapped for button: {btn_name}")

    def _needs_config(self, page) -> bool:
        parser = getattr(getattr(self, "page_config", None), "yaml_parser", None)
        return parser is not None and needs_config(page.objectName().removeprefix("page_"), parser.is_loaded())

    def _warn_no_config(self):
        CustomMessageBox.warning(
            "Load a config before recording: the Record page needs one to know which topics to show and record.\n\n"
            "Opening the Configuration page: insert a file, or build one with \"Build from ROS\".",
            title="No config loaded",
            parent=self,
        )

    def _update_watcher_path(self, theme):
        """Atualiza o ficheiro que o watcher está a monitorizar."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        qss_path = os.path.join(script_dir, "ui", "styles", f"relayd_{theme}.qss")

        # Remove caminhos antigos para evitar conflitos
        current_paths = self.qss_watcher.files()
        if current_paths:
            self.qss_watcher.removePaths(current_paths)

        if os.path.exists(qss_path):
            self.qss_watcher.addPath(qss_path)
            self.current_theme_file = qss_path

    def _on_qss_changed(self, path):
        """Callback disparado quando o ficheiro QSS é guardado no VS Code."""
        logger.info(f"Alteração detetada no QSS: {path}")
        try:
            with open(path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            logger.error(f"Erro ao recarregar QSS: {e}")

    def load_theme(self, theme="light"):
        """Load external QSS theme. Options: 'light', 'dark'."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        qss_path = os.path.join(script_dir, "ui", "styles", f"relayd_{theme}.qss")
        try:
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())
            # Atualiza o watcher sempre que o tema muda manualmente
            if hasattr(self, "qss_watcher"):
                self._update_watcher_path(theme)
        except FileNotFoundError:
            logger.warning(f"Theme file not found: {qss_path}")

    def _on_settings_changed(self, settings):
        """Apply the settings that take effect live: theme, ROS spin interval, log level."""
        changes = live_changes(settings, self._theme, self.ros_spin_timer.interval())
        if "theme" in changes:
            self._theme = changes["theme"]
            self.load_theme(self._theme)
        if "spin_ms" in changes:
            self.ros_spin_timer.setInterval(changes["spin_ms"])
        apply_user_level()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "resize_timer") and self.resize_timer is not None:
            self.resize_timer.stop()
            self.resize_timer.start(300)
        else:
            self.on_resize_finished()

    def on_resize_finished(self):
        try:
            if hasattr(self, "page_record") and self.page_record is not None:
                if hasattr(self.page_record, "on_window_resize"):
                    self.page_record.on_window_resize()
        except Exception as e:
            logger.info(f"Error handling resize: {e}")

    def closeEvent(self, event):
        """Clean up timers and ROS connections before closing"""
        if not self._recording_saved_or_dropped():
            event.ignore()
            return
        try:
            if getattr(self, "page_run_ui", None) is not None:  # a policy driving the robot stops first
                self.page_run_ui.shutdown()

            if hasattr(self, "page_record") and self.page_record is not None:
                # Stop TF update timerexit_code
                if hasattr(self.page_record, "tf_update_timer"):
                    self.page_record.tf_update_timer.stop()

                # Stop visualization timers
                if hasattr(self.page_record, "container_visualizer"):
                    for timer in self.page_record.container_visualizer.visualization_timers.values():
                        timer.stop()

                # Stop the TF listener's dedicated spin thread
                if hasattr(self.page_record, "topic_subscribers"):
                    self.page_record.topic_subscribers.close()

            # Stop any active rviz2 playback (DataPlayer + robot_state_publisher/rviz2)
            if hasattr(self, "page_review_ui") and self.page_review_ui is not None:
                if hasattr(self.page_review_ui, "shutdown_playback"):
                    self.page_review_ui.shutdown_playback()

            logger.info("Cleaned up timers and connections")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        event.accept()


    def _recording_saved_or_dropped(self) -> bool:
        """Stop and save a running recording. False (keep the window open) if it could not be saved
        and the user does not want to lose it: that demo is only in memory."""
        recorder = getattr(getattr(self, "page_record", None), "data_recorder", None)
        if recorder is None:
            return True
        try:
            if recorder.is_recording:
                recorder.stop_recording()
            with recorder.recording_lock:  # a Stop pressed just before may still be saving in the background: wait
                pass
            if not getattr(recorder, "_unsaved", False) or recorder.retry_save():
                return True
        except Exception as e:
            logger.error(f"Error saving the recording on close: {e}")
        return CustomMessageBox.question(
            f"The last recording could not be saved to {recorder.h5_file_path} (see the log).\n"
            "Close anyway and lose it?",
            title="Recording not saved",
        )


def main(args=None):
    domain = ros_domain_to_set(user_settings.values(), os.environ)  # the environment, if set, wins
    if domain is not None:
        os.environ["ROS_DOMAIN_ID"] = domain
    try:  # ROS must not take Ctrl+C for itself: the window closes on it instead (see ctrl_c.py)
        from rclpy.signals import SignalHandlerOptions

        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except ImportError:  # an rclpy without that option
        rclpy.init(args=args)

    node = Node("qt_visualizer_node")

    logger.info("DEBUG: About to create QApplication")
    app = QApplication(sys.argv)
    logger.info("DEBUG: QApplication created, about to create UI")
    window = UI(node)
    close_on_ctrl_c(window)
    logger.info("DEBUG: UI created, about to call app.exec_()")
    exit_code = app.exec_()
    logger.info(f"DEBUG: app.exec_() returned {exit_code}")

    # Clean up ROS
    try:
        rclpy.shutdown()
    except Exception:
        pass

    window.raise_()
    window.activateWindow()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
