import sys
import os

import rclpy
from rclpy.node import Node

from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QTextEdit, QPushButton
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QFileSystemWatcher, QObject, pyqtSignal

try:
    from relay_d.utils.coloring_logger import logger
    from relay_d.utils.coloring_logger import set_level
    set_level("INFO")   # or "DEBUG" / "ERROR" / etc.
except ImportError:
    # Fallback to standard logging if relay_d isn't installed/importable
    import logging
    logger = logging.getLogger(__name__)

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
except ImportError as e:
    logger.error(f"Failed to import one or more UI pages: {e}")
    ConfigPage = RecordPage = PostProcessPage = SettingsPage = ReviewPage = None


class UI(QMainWindow):
    def __init__(self, node):
        super(UI, self).__init__()
        self.node = node
        self.ros_nodes = [node]

        # ROS spin timer - this is critical for receiving ROS messages
        self.ros_spin_timer = QTimer()
        self.ros_spin_timer.timeout.connect(self.spin_once)
        self.ros_spin_timer.start(10)  # Spin every 10ms (100 Hz)

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

        self.load_theme("light")

        # Configuração do Live Reload para QSS
        self.qss_watcher = QFileSystemWatcher()
        self._update_watcher_path("light")
        self.qss_watcher.fileChanged.connect(self._on_qss_changed)

        try:
            self.btn_theme_toggle.clicked.connect(self._toggle_theme)
        except AttributeError:
            pass

        # Load the settings
        if hasattr(self, "page_settings_ui"):
            self.page_settings_ui.load_settings()

        # Change the current widget to the start page
        try:
            self.stackedWidget.setCurrentWidget(self.page_help)
        except Exception as e:
            logger.error(f"Error setting initial page: {e}")

        self.show()

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
                lock = getattr(node, "_spin_lock", None)
                if lock is not None:
                    with lock:
                        rclpy.spin_once(node, timeout_sec=0)
                else:
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
        }
        if btn_name in page_mapping:
            try:
                if btn_name == "btn_review" and hasattr(self, "page_review_ui"):
                    self.page_review_ui.sync_with_current_session()
                    self.page_review_ui.refresh_demo_list()
                self.stackedWidget.setCurrentWidget(page_mapping[btn_name])
            except Exception as e:
                logger.error(f"Error switching to page for {btn_name}: {e}")
        else:
            logger.warning(f"No page mapped for button: {btn_name}")

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

    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        current_style = self.btn_theme_toggle.text()
        if current_style == "Light":
            new_theme = "dark"
            self.btn_theme_toggle.setText("Dark")
        else:
            new_theme = "light"
            self.btn_theme_toggle.setText("Light")

        self.load_theme(new_theme)

    def on_menu_toggled(self, expanded):
        logger.info(f"Menu toggled: {'Expanded' if expanded else 'Collapsed'}")
        try:
            if hasattr(self, "page_record") and self.page_record is not None:
                if hasattr(self.page_record, "on_window_resize"):
                    self.page_record.on_window_resize()
        except Exception as e:
            logger.info(f"Error in menu toggle resize: {e}")

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

    def add_container_layout_controls(self):
        if hasattr(self, "btn_auto_arrange"):
            self.btn_auto_arrange.clicked.connect(self.auto_arrange_containers)

        if hasattr(self, "btn_2x1_layout"):
            self.btn_2x1_layout.clicked.connect(lambda: self.set_forced_layout(2, 1))

        if hasattr(self, "btn_2x2_layout"):
            self.btn_2x2_layout.clicked.connect(lambda: self.set_forced_layout(2, 2))

        if hasattr(self, "btn_3x2_layout"):
            self.btn_3x2_layout.clicked.connect(lambda: self.set_forced_layout(3, 2))

    def auto_arrange_containers(self):
        try:
            if hasattr(self, "page_record") and self.page_record is not None:
                if hasattr(self.page_record, "rearrange_containers"):
                    self.page_record.rearrange_containers()
        except Exception as e:
            logger.info(f"Error in auto arrange: {e}")

    def set_forced_layout(self, cols, rows):
        try:
            if hasattr(self, "page_record") and self.page_record is not None:
                if hasattr(self.page_record, "set_container_grid_layout"):
                    self.page_record.set_container_grid_layout(cols, rows)
        except Exception as e:
            logger.info(f"Error in forced layout: {e}")

    def closeEvent(self, event):
        """Clean up timers and ROS connections before closing"""
        try:
            if hasattr(self, "page_record") and self.page_record is not None:
                # Stop TF update timerexit_code
                if hasattr(self.page_record, "tf_update_timer"):
                    self.page_record.tf_update_timer.stop()

                # Stop visualization timers
                if hasattr(self.page_record, "container_visualizer"):
                    for timer in self.page_record.container_visualizer.visualization_timers.values():
                        timer.stop()

                # Stop data recorder
                if hasattr(self.page_record, "data_recorder"):
                    if self.page_record.data_recorder.is_recording:
                        self.page_record.data_recorder.stop_recording()

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


def main(args=None):
    rclpy.init(args=args)

    node = Node("qt_visualizer_node")

    logger.info("DEBUG: About to create QApplication")
    app = QApplication(sys.argv)
    logger.info("DEBUG: QApplication created, about to create UI")
    window = UI(node)
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
