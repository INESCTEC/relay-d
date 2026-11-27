"""Creates and lays out live sensor container cards on the Record page."""

from __future__ import annotations

import math
import os

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from relay_d.utils.coloring_logger import logger

_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_UI = os.path.join(_UI_DIR, "imageWidget.ui")
_UI_FILE_MAP = {
    "image": _DEFAULT_UI,
    "joint": os.path.join(_UI_DIR, "jointWidget.ui"),
    "pose": os.path.join(_UI_DIR, "poseWidget.ui"),
    "tf": os.path.join(_UI_DIR, "poseWidget.ui"),
    "gripper": os.path.join(_UI_DIR, "actuatorWidget.ui"),
}


def _grid_dimensions(count: int) -> tuple[int, int]:
    """Return ``(cols, rows)`` for arranging ``count`` container cards cleanly."""
    if count <= 1:
        return (1, 1)
    if count <= 3:
        return (count, 1)
    if count <= 4:
        return (2, 2)
    if count <= 6:
        return (3, 2)
    if count <= 9:
        return (3, 3)
    if count <= 12:
        return (4, 3)
    cols = math.ceil(math.sqrt(count))
    return (cols, math.ceil(count / cols))


class TopicContainers:
    """Manages the grid of `.ui` sensor cards shown on the Record page."""

    def __init__(self, ui_instance):
        self.ui = ui_instance
        self.containers = []
        self.container_count = 0

    def _get_ui_file_for_topic_type(self, topic_type: str) -> str:
        ui_file = _UI_FILE_MAP.get((topic_type or "").lower(), _DEFAULT_UI)
        if not os.path.exists(ui_file):
            logger.warning(f"UI file {ui_file} not found, falling back to {_DEFAULT_UI}.")
            return _DEFAULT_UI
        return ui_file

    def create_container(self, topic_type, container_data=None):
        """Instantiate a container card widget from its `.ui` template and attach metadata."""
        try:
            ui_file = self._get_ui_file_for_topic_type(topic_type)
            container = QWidget()
            uic.loadUi(ui_file, container)

            container.topic_type = topic_type
            input_name = container_data.get("input_name") if container_data else None
            if input_name:
                container.container_id = input_name
            else:
                container.container_id = f"container_{self.container_count}_{topic_type}"
                self.container_count += 1

            container.setProperty("container_id", container.container_id)
            container.setProperty("class", "TopicContainer")
            container.style().unpolish(container)
            container.style().polish(container)

            if container_data:
                for key, value in container_data.items():
                    setattr(container, key, value)

            self.containers.append(container)
            return container
        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            return None

    def _ensure_grid_layout(self, target_widget):
        """Return (and lazily create) a scrollable QGridLayout hosted inside ``target_widget``."""
        if hasattr(target_widget, "_container_layout_setup"):
            return target_widget._container_layout_setup

        scroll_area = QScrollArea()
        scroll_area.setObjectName("container_scroll_area")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container_widget = QWidget()
        container_widget.setObjectName(f"{target_widget.objectName()}_containers")
        container_widget.setStyleSheet("background: transparent;")
        container_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid_layout = QGridLayout(container_widget)
        grid_layout.setSpacing(5)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        scroll_area.setWidget(container_widget)

        existing_layout = target_widget.layout()
        if existing_layout is None:
            main_layout = QVBoxLayout(target_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            main_layout.addWidget(scroll_area)
        else:
            existing_layout.addWidget(scroll_area)

        target_widget._container_scroll_area = scroll_area
        target_widget._container_widget = container_widget
        target_widget._container_layout_setup = grid_layout
        return grid_layout

    @staticmethod
    def _apply_size_policy(container):
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container.setMinimumSize(120, 100)
        container.setMaximumSize(16777215, 16777215)

    def _resolve_target(self, target_widget):
        return target_widget if target_widget is not None else getattr(self.ui, "page_start", self.ui)

    def add_container_to_widget(self, container, target_widget=None, *, cols: int = 0, index: int = -1):
        """Place ``container`` into ``target_widget``'s grid layout."""
        if container is None:
            logger.error("Cannot add None container")
            return False
        try:
            target = self._resolve_target(target_widget)
            grid_layout = self._ensure_grid_layout(target)
            pos = index if index >= 0 else grid_layout.count()
            n_cols = cols if cols > 0 else _grid_dimensions(pos + 1)[0]
            grid_layout.addWidget(container, pos // n_cols, pos % n_cols)
            self._apply_size_policy(container)
            container.show()
            return True
        except Exception as e:
            logger.error(f"Failed to add container to widget: {e}")
            return False

    def create_and_add_container(self, topic_type, container_data=None, target_widget=None):
        container = self.create_container(topic_type, container_data)
        if container and self.add_container_to_widget(container, target_widget):
            return container
        return None

    def create_multiple_containers(self, container_specs, target_widget=None):
        """Create and lay out all containers in one grid sized for ``len(container_specs)``."""
        target = self._resolve_target(target_widget)
        grid_layout = self._ensure_grid_layout(target)
        valid_specs = [spec for spec in container_specs if "topic_type" in spec]
        cols, rows = _grid_dimensions(len(valid_specs))

        created = []
        for idx, spec in enumerate(valid_specs):
            container = self.create_container(spec["topic_type"], spec.get("data", {}))
            if container and self.add_container_to_widget(container, target, cols=cols, index=idx):
                created.append(container)

        for col in range(cols):
            grid_layout.setColumnStretch(col, 1)
        for row in range(rows):
            grid_layout.setRowStretch(row, 1)

        if hasattr(target, "_container_widget"):
            target._container_widget.updateGeometry()
        target.update()
        logger.info(f"Created {len(created)}/{len(container_specs)} containers ({cols}x{rows} grid)")
        return created

    def update_container_info(self, container, info_dict):
        """Update container title/tooltip/status labels without overwriting metric badges."""
        try:
            name_label = next(
                (getattr(container, n) for n in ("label_topic_name", "text_data_display", "label_actuator_name")
                 if hasattr(container, n)),
                None,
            )
            if name_label is not None and info_dict.get("input_name"):
                name_label.setText(str(info_dict["input_name"]))
                if info_dict.get("topic_path"):
                    name_label.setToolTip(str(info_dict["topic_path"]))

            if "status" in info_dict:
                status = str(info_dict["status"])
                area = next(
                    (getattr(container, n) for n in ("label_actuator_state", "dataVisualizationLabel")
                     if hasattr(container, n)),
                    None,
                )
                if area is not None:
                    area.setText(status)
                container.setToolTip(status)
        except Exception as e:
            logger.error(f"Failed to update container info: {e}")

    def clear_all_containers(self):
        """Remove and delete all tracked container widgets."""
        for container in list(self.containers):
            try:
                parent = container.parent()
                if parent and parent.layout():
                    parent.layout().removeWidget(container)
                container.hide()
                container.deleteLater()
            except Exception as e:
                logger.error(f"Failed to remove container: {e}")
        self.containers.clear()
        self.container_count = 0

    def auto_arrange_containers(self):
        for container in self.containers:
            self._apply_size_policy(container)
            container.updateGeometry()

    def resize_containers_to_window(self, window_size=None):
        """Update container grid geometry after window resize / fullscreen transitions."""
        if not self.containers:
            return
        try:
            page = getattr(self.ui, "page_start", None)
            for container in self.containers:
                self._apply_size_policy(container)
                container.updateGeometry()
            if page is not None and hasattr(page, "_container_layout_setup"):
                page._container_layout_setup.invalidate()
                page._container_layout_setup.activate()
                if hasattr(page, "_container_widget"):
                    page._container_widget.updateGeometry()
            self.ui.update()
        except Exception as e:
            logger.error(f"Failed to resize containers: {e}")

    def reset_layout_references(self, target_widget):
        """Detach and remove the container scroll area from ``target_widget``."""
        if target_widget is None:
            return
        try:
            if hasattr(target_widget, "_container_layout_setup"):
                delattr(target_widget, "_container_layout_setup")
            if hasattr(target_widget, "_container_scroll_area"):
                scroll_area = target_widget._container_scroll_area
                if scroll_area is not None:
                    parent = scroll_area.parent()
                    if parent is not None and parent.layout() is not None:
                        parent.layout().removeWidget(scroll_area)
                    scroll_area.deleteLater()
                delattr(target_widget, "_container_scroll_area")
            if hasattr(target_widget, "_container_widget"):
                delattr(target_widget, "_container_widget")
        except Exception as e:
            logger.error(f"Error resetting layout references: {e}")
