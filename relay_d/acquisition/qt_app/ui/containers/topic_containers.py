import os
import sys
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QTextEdit,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QFrame,
    QGridLayout,
    QSizePolicy,
)
from PyQt5 import uic
from PyQt5.QtCore import Qt
from relay_d.utils.coloring_logger import logger


# Class used to create containers for the selected topics
class TopicContainers:
    def __init__(self, ui_instance):
        self.ui = ui_instance
        self.containers = []  # Keep track of created containers
        self.container_count = 0  # For unique naming

        # Initialize the container area if it doesn't exist
        self._setup_container_area()

    def _setup_container_area(self):
        """
        Set up a scroll area and layout for containers if not already present
        """
        # Check if the UI already has a designated area for containers
        if not hasattr(self.ui, "container_scroll_area"):
            # Create a scroll area for containers
            self.ui.container_scroll_area = QScrollArea()
            self.ui.container_scroll_area.setWidgetResizable(True)
            self.ui.container_scroll_area.setHorizontalScrollBarPolicy(
                Qt.ScrollBarAsNeeded
            )
            self.ui.container_scroll_area.setVerticalScrollBarPolicy(
                Qt.ScrollBarAsNeeded
            )

            # Create a widget to hold all containers
            self.ui.container_widget = QWidget()
            # Start with a grid layout
            self.ui.container_layout = QGridLayout(self.ui.container_widget)
            self.ui.container_layout.setSpacing(10)  # Add spacing between containers
            self.ui.container_layout.setContentsMargins(10, 10, 10, 10)  # Add margins

            # Set the widget to the scroll area
            self.ui.container_scroll_area.setWidget(self.ui.container_widget)

    def create_container(self, topic_type, container_data=None):
        """
        Create a container widget from the UI file

        Args:
            topic_type (str): Type of topic (e.g., 'image', 'pointcloud', etc.)
            container_data (dict, optional): Additional data to set on the container

        Returns:
            QWidget: The loaded container widget
        """
        try:
            # Determine which UI file to load based on topic type
            ui_file = self._get_ui_file_for_topic_type(topic_type)

            container = QWidget()
            uic.loadUi(ui_file, container)

            # Set basic properties for identification
            container.topic_type = topic_type

            # Use output_map from container_data if available, otherwise use input_name
            output_map = None
            input_name = None
            if container_data:
                output_map = container_data.get("output_map")
                input_name = container_data.get("input_name")

            # Use output_map as container_id (this is the name used in HDF5 output)
            if output_map:
                container.container_id = output_map
            elif input_name:
                container.container_id = input_name
            else:
                container.container_id = f"container_{self.container_count}_{topic_type}"
                self.container_count += 1

            # --- BLOCO PARA QSS PERSONALIZADO ---
            # Define o ID para seletores específicos: [container_id="valor"]
            container.setProperty("container_id", container.container_id)
            # Define a Classe para seletor comum: QWidget.TopicContainer
            container.setProperty("class", "TopicContainer")

            # Forçar o Qt a atualizar o estilo com as novas propriedades
            container.style().unpolish(container)
            container.style().polish(container)
            # -------------------------------------

            # Set additional data if provided
            if container_data:
                for key, value in container_data.items():
                    setattr(container, key, value)

            # Add to our tracking list
            self.containers.append(container)

            logger.info(f"Created container for topic type: {topic_type}")
            return container

        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            return None

    def _get_ui_file_for_topic_type(self, topic_type):
        """
        Get the appropriate UI file based on topic type

        Args:
            topic_type (str): The topic type

        Returns:
            str: Path to the UI file
        """
        # Get the directory where this Python file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Map topic types to UI files
        ui_file_map = {
            "image": os.path.join(current_dir, "imageWidget.ui"),
            "pointcloud": os.path.join(current_dir, "pointcloudWidget.ui"),
            "joint": os.path.join(current_dir, "jointWidget.ui"),
            "tool": os.path.join(current_dir, "toolWidget.ui"),
            "pose": os.path.join(current_dir, "poseWidget.ui"),
            "tf": os.path.join(current_dir, "poseWidget.ui"),
            "gripper": os.path.join(current_dir, "actuactorWidget.ui"),
        }

        # Get the UI file for the topic type, default to imageWidget.ui
        ui_file = ui_file_map.get(
            topic_type.lower(), os.path.join(current_dir, "imageWidget.ui")
        )

        # Check if file exists, if not use default
        if not os.path.exists(ui_file):
            logger.warning(
                f"UI file {ui_file} not found, skipping the use of container template."
            )

        return ui_file

    def add_container_to_widget(self, container, target_widget=None):
        """
        Add a container to a target widget or the default container area

        Args:
            container (QWidget): The container to add
            target_widget (QWidget, optional): Target widget to add to.
                                             If None, uses default container area
        """
        if container is None:
            logger.error("Cannot add None container")
            return False

        try:
            if target_widget is not None:
                # Set up layout within the target widget
                target_layout = self._setup_target_widget_layout(target_widget)
                if target_layout is None:
                    logger.error("Failed to set up target widget layout")
                    return False

                self._add_to_grid_layout(container, target_layout)

                # Set correct parent - use the container widget if available
                if hasattr(target_widget, "_container_widget"):
                    container.setParent(target_widget._container_widget)
                else:
                    container.setParent(target_widget)

                logger.info(
                    f"Added container to target widget: {target_widget.objectName()}"
                )

            else:
                # Add to default container area using grid layout
                self._add_to_grid_layout(container, self.ui.container_layout)
                container.setParent(self.ui.container_widget)
                logger.info("Added container to default container area")

            # Set container size policy and constraints
            self._set_container_size_constraints(container)

            # Make container visible
            container.show()

            # Force layout update
            self._force_layout_update(target_widget)

            return True

        except Exception as e:
            logger.error(f"Failed to add container to widget: {e}")
            return False

    def _force_layout_update(self, target_widget):
        """Force update of layout and widget display"""
        try:
            if target_widget:
                # Update container widget if it exists
                if hasattr(target_widget, "_container_widget"):
                    target_widget._container_widget.adjustSize()
                    target_widget._container_widget.update()
                    target_widget._container_widget.repaint()

                # Update scroll area if it exists
                if hasattr(target_widget, "_container_scroll_area"):
                    target_widget._container_scroll_area.update()
                    target_widget._container_scroll_area.repaint()

                # Update target widget
                target_widget.update()
                target_widget.repaint()

                logger.info(f"Forced layout update for {target_widget.objectName()}")
        except Exception as e:
            logger.error(f"Error in force layout update: {e}")

    def _setup_target_widget_layout(self, target_widget):
        """
        Set up or get the layout for the target widget - STRETCHING VERSION

        Args:
            target_widget (QWidget): The target widget

        Returns:
            QGridLayout: The layout for containers
        """
        # Check if we already set up a container layout for this widget
        if hasattr(target_widget, "_container_layout_setup"):
            return target_widget._container_layout_setup

        # Check if target widget already has a layout
        existing_layout = target_widget.layout()

        if existing_layout is None:
            # Create a scroll area to ensure all containers are visible
            scroll_area = QScrollArea(target_widget)
            scroll_area.setObjectName("container_scroll_area")
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            # Create container widget
            container_widget = QWidget()
            container_widget.setObjectName("container_widget")
            container_widget.setStyleSheet("background: transparent;")

            # Set size policy for container widget to expand
            container_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            # Create grid layout for containers
            grid_layout = QGridLayout(container_widget)
            grid_layout.setSpacing(5)  # Reduced spacing for better space utilization
            grid_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins

            # Set the grid layout to expand columns and rows evenly
            grid_layout.setColumnStretch(0, 1)
            grid_layout.setColumnStretch(1, 1)
            grid_layout.setColumnStretch(2, 1)
            grid_layout.setRowStretch(0, 1)
            grid_layout.setRowStretch(1, 1)

            # Set up scroll area
            scroll_area.setWidget(container_widget)

            # Add scroll area to target widget
            main_layout = QVBoxLayout(target_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            main_layout.addWidget(scroll_area)

            # Store references
            target_widget._container_scroll_area = scroll_area
            target_widget._container_widget = container_widget
            target_widget._container_layout_setup = grid_layout

            logger.info(
                f"Created stretching container area for {target_widget.objectName()}"
            )
            return grid_layout

        elif isinstance(existing_layout, QGridLayout):
            # Use existing grid layout but ensure it's set up for stretching
            existing_layout.setSpacing(5)
            existing_layout.setContentsMargins(5, 5, 5, 5)

            # Set column and row stretches
            for col in range(existing_layout.columnCount()):
                existing_layout.setColumnStretch(col, 1)
            for row in range(existing_layout.rowCount()):
                existing_layout.setRowStretch(row, 1)

            target_widget._container_layout_setup = existing_layout
            logger.info(
                f"Updated existing grid layout for stretching: {target_widget.objectName()}"
            )
            return existing_layout

        else:
            # Target widget has a different layout type
            # Create a scroll area within the existing layout

            # Check if there's already a container scroll area
            scroll_area = None
            for i in range(existing_layout.count()):
                item = existing_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QScrollArea):
                    if hasattr(item.widget(), "_is_container_scroll_area"):
                        scroll_area = item.widget()
                        break

            if scroll_area is None:
                # Create new scroll area
                scroll_area = QScrollArea()
                scroll_area._is_container_scroll_area = True
                scroll_area.setWidgetResizable(True)
                scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

                # Create container widget
                container_widget = QWidget()
                container_widget.setObjectName(
                    f"{target_widget.objectName()}_containers"
                )
                container_widget.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding
                )

                # Create grid layout
                grid_layout = QGridLayout(container_widget)
                grid_layout.setSpacing(5)
                grid_layout.setContentsMargins(5, 5, 5, 5)

                # Set stretching for grid
                grid_layout.setColumnStretch(0, 1)
                grid_layout.setColumnStretch(1, 1)
                grid_layout.setColumnStretch(2, 1)
                grid_layout.setRowStretch(0, 1)
                grid_layout.setRowStretch(1, 1)

                # Set up scroll area
                scroll_area.setWidget(container_widget)

                # Add to existing layout
                existing_layout.addWidget(scroll_area)

                # Store references
                target_widget._container_scroll_area = scroll_area
                target_widget._container_widget = container_widget
                target_widget._container_layout_setup = grid_layout

                logger.info(
                    f"Created stretching container scroll area within existing layout for {target_widget.objectName()}"
                )
                return grid_layout
            else:
                # Use existing scroll area's layout
                container_widget = scroll_area.widget()
                if container_widget and container_widget.layout():
                    grid_layout = container_widget.layout()
                    # Update for stretching
                    grid_layout.setSpacing(5)
                    grid_layout.setContentsMargins(5, 5, 5, 5)

                    # Set stretching
                    for col in range(grid_layout.columnCount()):
                        grid_layout.setColumnStretch(col, 1)
                    for row in range(grid_layout.rowCount()):
                        grid_layout.setRowStretch(row, 1)

                    target_widget._container_layout_setup = grid_layout
                    return grid_layout
                else:
                    logger.error(
                        "Existing scroll area has no container widget or layout"
                    )
                    return None

    def _add_to_grid_layout(self, container, grid_layout):
        """Add container to grid layout in matrix format - FIXED OVERLAPPING ISSUE"""
        # Get current count of widgets already in the layout
        current_count = 0

        # Count all non-null items in the grid layout
        # We need to check a larger range to account for sparse grids
        max_check_rows = max(grid_layout.rowCount(), 10)
        max_check_cols = max(grid_layout.columnCount(), 10)

        # Debug: log current grid state
        logger.info(
            f"Grid state before adding: rows={grid_layout.rowCount()}, cols={grid_layout.columnCount()}"
        )

        # Build a map of occupied positions
        occupied_positions = set()

        for i in range(max_check_rows):
            for j in range(max_check_cols):
                item = grid_layout.itemAtPosition(i, j)
                if item is not None and item.widget() is not None:
                    current_count += 1
                    occupied_positions.add((i, j))
                    logger.info(f"Found existing widget at ({i}, {j})")

        logger.info(f"Current widget count in grid: {current_count}")
        logger.info(f"Occupied positions: {occupied_positions}")

        # Calculate grid dimensions for the total count including this new container
        total_count = current_count + 1
        cols, rows = self._calculate_grid_dimensions(total_count)

        logger.info(
            f"Calculated grid dimensions for {total_count} containers: {cols}x{rows}"
        )

        # Find the first available position in the grid
        target_row = None
        target_col = None

        # Search for available position in row-major order
        for row in range(rows):
            for col in range(cols):
                if (row, col) not in occupied_positions:
                    target_row = row
                    target_col = col
                    break
            if target_row is not None:
                break

        # Fallback: if no position found, use the calculated position
        if target_row is None or target_col is None:
            target_row = current_count // cols
            target_col = current_count % cols
            logger.warning(
                f"Using fallback position calculation: ({target_row}, {target_col})"
            )

        # Double-check that the position is not already occupied
        while (target_row, target_col) in occupied_positions:
            target_col += 1
            if target_col >= cols:
                target_col = 0
                target_row += 1
            logger.info(
                f"Position ({target_row}, {target_col}) occupied, trying next..."
            )

        # Add container to grid
        grid_layout.addWidget(container, target_row, target_col)

        logger.info(
            f"Added container to grid at position ({target_row}, {target_col}) - Total containers: {total_count}"
        )

        # Verify the widget was actually added
        added_item = grid_layout.itemAtPosition(target_row, target_col)
        if added_item and added_item.widget():
            logger.info(
                f"Confirmed: Widget successfully added at ({target_row}, {target_col})"
            )
        else:
            logger.error(f"Widget was not added at ({target_row}, {target_col})")

    def _calculate_grid_dimensions(self, container_count):
        """
        Calculate optimal grid dimensions based on container count - IMPROVED VERSION

        Args:
            container_count (int): Number of containers

        Returns:
            tuple: (columns, rows)
        """
        if container_count <= 0:
            return (1, 1)
        elif container_count == 1:
            return (1, 1)
        elif container_count == 2:
            return (2, 1)
        elif container_count == 3:
            return (3, 1)  # Display 3 containers in a single row
        elif container_count <= 4:
            return (2, 2)
        elif container_count == 5:
            return (3, 2)  # 3 columns, 2 rows for 5 containers
        elif container_count <= 6:
            return (3, 2)
        elif container_count <= 9:
            return (3, 3)
        elif container_count <= 12:
            return (4, 3)
        elif container_count <= 16:
            return (4, 4)
        else:
            # For larger numbers, try to make it as square as possible
            import math

            cols = math.ceil(math.sqrt(container_count))
            rows = math.ceil(container_count / cols)
            return (cols, rows)

    def _set_container_size_constraints(self, container):
        """Set size policies and constraints for containers - STRETCHING VERSION"""
        # Set size policy to allow containers to expand and fill available space
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Set minimum size to ensure containers are readable
        container.setMinimumSize(120, 100)

        # Remove maximum size constraints to allow unlimited expansion
        container.setMaximumSize(16777215, 16777215)  # Maximum possible size in Qt

        # Set preferred size - this will be the starting point for expansion
        container.resize(200, 150)

    def clear_all_containers(self):
        """
        Remove all containers - IMPROVED VERSION
        """
        logger.info(f"Clearing {len(self.containers)} containers")

        # Create a copy of the containers list to avoid modification during iteration
        containers_to_remove = self.containers.copy()

        for container in containers_to_remove:
            try:
                # Remove from parent widget's layout
                parent = container.parent()
                if parent and parent.layout():
                    parent.layout().removeWidget(container)

                # Hide the container first
                container.hide()

                # Remove from tracking list
                if container in self.containers:
                    self.containers.remove(container)

                # Delete the container
                container.deleteLater()

                logger.info(
                    f"Removed container: {getattr(container, 'container_id', 'unknown')}"
                )

            except Exception as e:
                logger.error(f"Failed to remove container: {e}")

        # Clear the containers list
        self.containers.clear()

        # Reset container count
        self.container_count = 0

        logger.info("Cleared all containers")

    def remove_container(self, container):
        """
        Remove a container from the UI and tracking list

        Args:
            container (QWidget): The container to remove
        """
        try:
            if container in self.containers:
                self.containers.remove(container)

            # Remove from parent widget's layout
            parent = container.parent()
            if parent and parent.layout():
                parent.layout().removeWidget(container)

            # Hide and delete the container
            container.hide()
            container.deleteLater()

            logger.info(
                f"Removed container: {getattr(container, 'container_id', 'unknown')}"
            )

        except Exception as e:
            logger.error(f"Failed to remove container: {e}")

    def get_containers_by_type(self, topic_type):
        """
        Get all containers of a specific topic type

        Args:
            topic_type (str): The topic type to filter by

        Returns:
            list: List of containers matching the topic type
        """
        return [
            container
            for container in self.containers
            if hasattr(container, "topic_type") and container.topic_type == topic_type
        ]

    def create_and_add_container(
        self, topic_type, container_data=None, target_widget=None
    ):
        """
        Convenience method to create and add a container in one call

        Args:
            topic_type (str): Type of topic
            container_data (dict, optional): Additional data to set on the container
            target_widget (QWidget, optional): Target widget to add to

        Returns:
            QWidget: The created and added container, or None if failed
        """
        container = self.create_container(topic_type, container_data)
        if container and self.add_container_to_widget(container, target_widget):
            return container
        return None

    def update_container_info(self, container, info_dict):
        """
        Update container UI elements with provided information

        Args:
            container (QWidget): The container to update
            info_dict (dict): Dictionary with information to update
        """
        try:
            topic_type = getattr(container, "topic_type", "unknown")

            if topic_type == "image":
                image_labels = {
                    "input_name": "label_topic_name",
                    "topic_path": "label_topic_name",
                    "status": "label_frame_rate",
                }
                for key, label_name in image_labels.items():
                    if key in info_dict and hasattr(container, label_name):
                        container.__getattribute__(label_name).setText(
                            str(info_dict[key])
                        )

            elif topic_type in ["joint", "pose", "tf"]:
                if "input_name" in info_dict and hasattr(
                    container, "text_data_display"
                ):
                    container.text_data_display.setText(str(info_dict["input_name"]))
                if "status" in info_dict and hasattr(container, "positionLabel"):
                    container.positionLabel.setText(str(info_dict["status"]))

            elif topic_type == "gripper":
                if "input_name" in info_dict and hasattr(
                    container, "label_actuator_name"
                ):
                    container.label_actuator_name.setText(str(info_dict["input_name"]))
                if "status" in info_dict and hasattr(container, "label_actuator_state"):
                    container.label_actuator_state.setText(str(info_dict["status"]))

            else:
                for key, value in info_dict.items():
                    if hasattr(container, f"label_{key}"):
                        getattr(container, f"label_{key}").setText(str(value))

        except Exception as e:
            logger.error(f"Failed to update container info: {e}")

    def get_container_count(self):
        """Get the number of containers"""
        return len(self.containers)

    def get_container_by_id(self, container_id):
        """Get container by its ID"""
        for container in self.containers:
            if (
                hasattr(container, "container_id")
                and container.container_id == container_id
            ):
                return container
        return None

    def get_container_by_attribute(self, attribute_name, attribute_value):
        """
        Get container by any attribute

        Args:
            attribute_name (str): Name of the attribute to search by
            attribute_value: Value to match

        Returns:
            QWidget or None: The first container matching the criteria
        """
        for container in self.containers:
            if (
                hasattr(container, attribute_name)
                and getattr(container, attribute_name) == attribute_value
            ):
                return container
        return None

    def get_containers_by_attribute(self, attribute_name, attribute_value):
        """
        Get all containers matching an attribute

        Args:
            attribute_name (str): Name of the attribute to search by
            attribute_value: Value to match

        Returns:
            list: List of containers matching the criteria
        """
        return [
            container
            for container in self.containers
            if hasattr(container, attribute_name)
            and getattr(container, attribute_name) == attribute_value
        ]

    def update_container_status(self, container_id, status):
        """
        Update the status of a specific container by ID

        Args:
            container_id (str): The container ID
            status (str): The new status
        """
        container = self.get_container_by_id(container_id)
        if container:
            self.update_container_info(container, {"status": status})
            return True
        return False

    def get_all_container_info(self):
        """
        Get information about all containers

        Returns:
            list: List of dictionaries with container information
        """
        container_info = []
        for container in self.containers:
            info = {
                "container_id": getattr(container, "container_id", "unknown"),
                "topic_type": getattr(container, "topic_type", "unknown"),
            }

            # Add any additional attributes that exist
            for attr_name in ["input_name", "topic_path", "topic_type_msg", "status"]:
                if hasattr(container, attr_name):
                    info[attr_name] = getattr(container, attr_name)

            container_info.append(info)

        return container_info

    def create_multiple_containers(self, container_specs, target_widget=None):
        """
        Create multiple containers from a list of specifications - STRETCHING VERSION

        Args:
            container_specs (list): List of dictionaries with container specifications
                                   Each dict should have 'topic_type' and optionally 'data'
            target_widget (QWidget, optional): Target widget to add containers to

        Returns:
            list: List of successfully created containers
        """
        created_containers = []

        logger.info(f"Creating {len(container_specs)} containers with stretching...")

        # Set up the target widget layout first
        if target_widget is not None:
            target_layout = self._setup_target_widget_layout(target_widget)
            if target_layout is None:
                logger.error("Failed to set up target widget layout")
                return []

        # Calculate the final grid dimensions for all containers
        total_containers = len(container_specs)
        final_cols, final_rows = self._calculate_grid_dimensions(total_containers)
        logger.info(f"Final grid will be: {final_cols} columns x {final_rows} rows")

        # Create and add containers one by one
        for i, spec in enumerate(container_specs):
            if "topic_type" not in spec:
                logger.warning(f"Container spec {i} missing topic_type: {spec}")
                continue

            topic_type = spec["topic_type"]
            container_data = spec.get("data", {})

            logger.info(
                f"Creating container {i + 1}/{len(container_specs)}: {topic_type}"
            )

            # Create and add container
            container = self.create_and_add_container(
                topic_type, container_data, target_widget
            )
            if container:
                created_containers.append(container)
                logger.info(
                    f"Successfully created container {i + 1}: {container.container_id}"
                )
            else:
                logger.error(f"Failed to create container {i + 1}")

        # Now set up the grid layout for optimal stretching
        if target_widget is not None and hasattr(
            target_widget, "_container_layout_setup"
        ):
            grid_layout = target_widget._container_layout_setup

            # Set column and row stretches for the final grid
            for col in range(final_cols):
                grid_layout.setColumnStretch(col, 1)
                logger.info(f"Set column {col} stretch to 1")

            for row in range(final_rows):
                grid_layout.setRowStretch(row, 1)
                logger.info(f"Set row {row} stretch to 1")

            # Force layout update
            self._force_layout_update(target_widget)

            # Ensure all containers are visible and properly sized for stretching
            for container in created_containers:
                container.show()
                container.raise_()
                # Apply stretching size constraints
                self._set_container_size_constraints(container)

        logger.info(
            f"Successfully created {len(created_containers)}/{len(container_specs)} containers with stretching"
        )
        return created_containers

    def auto_arrange_containers(self):
        """
        Automatically arrange containers in optimal grid layout - IMPROVED VERSION
        """
        if not self.containers:
            return

        try:
            # Force update of all container parents
            for container in self.containers:
                if container.parent():
                    container.parent().update()
                    container.parent().repaint()
                container.update()
                container.repaint()
                # Ensure proper size constraints
                self._set_container_size_constraints(container)

            logger.info(f"Auto-arranged {len(self.containers)} containers")

        except Exception as e:
            logger.error(f"Failed to auto-arrange containers: {e}")

    def resize_containers_to_window(self, window_size=None):
        """
        Resize containers to fit optimally within the window - FULLSCREEN SUPPORT

        Args:
            window_size (QSize, optional): The window size. If None, tries to detect from UI
        """
        try:
            if window_size is None and hasattr(self.ui, "size"):
                window_size = self.ui.size()

            if not window_size or len(self.containers) == 0:
                return

            logger.info(
                f"Resizing containers for window size: {window_size.width()}x{window_size.height()}"
            )

            # Calculate available space for page_start widget
            if hasattr(self.ui, "page_start"):
                page_size = self.ui.page_start.size()
                available_width = max(page_size.width() - 20, 300)  # Reduced margins
                available_height = max(page_size.height() - 20, 200)
                logger.info(f"Available space: {available_width}x{available_height}")
            else:
                available_width = max(window_size.width() - 50, 400)
                available_height = max(window_size.height() - 100, 300)

            # Force layout update for all containers
            for container in self.containers:
                # Ensure containers can expand to fill space
                container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                container.setMinimumSize(100, 80)  # Smaller minimum for better scaling
                container.setMaximumSize(16777215, 16777215)  # No maximum limit

                # Force update
                container.updateGeometry()
                container.update()

            # Update grid layout stretching
            if hasattr(self.ui, "page_start") and hasattr(
                self.ui.page_start, "_container_layout_setup"
            ):
                grid_layout = self.ui.page_start._container_layout_setup
                container_widget = self.ui.page_start._container_widget

                # Update container widget size
                container_widget.resize(available_width, available_height)
                container_widget.updateGeometry()

                # Force grid layout update
                grid_layout.update()
                grid_layout.activate()

                logger.info(
                    f"Updated grid layout for {len(self.containers)} containers"
                )

            # Force UI refresh
            self._force_layout_update_all()

            logger.info(f"Resized containers for fullscreen/window resize")

        except Exception as e:
            logger.error(f"Failed to resize containers: {e}")

    def _force_layout_update_all(self):
        """Force comprehensive layout update for fullscreen/resize"""
        try:
            # Update all container widgets
            for container in self.containers:
                container.updateGeometry()
                container.update()
                container.repaint()

            # Update main UI components
            if hasattr(self.ui, "page_start"):
                # Update container widget
                if hasattr(self.ui.page_start, "_container_widget"):
                    self.ui.page_start._container_widget.updateGeometry()
                    self.ui.page_start._container_widget.update()
                    self.ui.page_start._container_widget.repaint()

                # Update scroll area
                if hasattr(self.ui.page_start, "_container_scroll_area"):
                    self.ui.page_start._container_scroll_area.updateGeometry()
                    self.ui.page_start._container_scroll_area.update()
                    self.ui.page_start._container_scroll_area.repaint()

                # Update page_start
                self.ui.page_start.updateGeometry()
                self.ui.page_start.update()
                self.ui.page_start.repaint()

            # Update main UI
            self.ui.updateGeometry()
            self.ui.update()
            self.ui.repaint()

            logger.info("Completed comprehensive layout update")

        except Exception as e:
            logger.error(f"Error in comprehensive layout update: {e}")

    def handle_window_resize(self, new_size):
        """Handle window resize events to expand containers"""
        try:
            logger.info(
                f"Handling window resize to: {new_size.width()}x{new_size.height()}"
            )
            self.resize_containers_to_window(new_size)
        except Exception as e:
            logger.error(f"Error handling window resize: {e}")

    def debug_containers(self):
        """Debug method to log container information - ENHANCED VERSION"""
        try:
            logger.info(f"=== Container Debug Info ===")
            logger.info(f"Total containers tracked: {len(self.containers)}")

            for i, container in enumerate(self.containers):
                container_id = getattr(container, "container_id", "unknown")
                is_visible = container.isVisible()
                size = container.size()
                pos = container.pos()
                parent = container.parent()
                parent_name = parent.objectName() if parent else "None"

                logger.info(f"Container {i + 1}: {container_id}")
                logger.info(f"  Visible: {is_visible}")
                logger.info(f"  Size: {size.width()}x{size.height()}")
                logger.info(f"  Position: ({pos.x()}, {pos.y()})")
                logger.info(f"  Parent: {parent_name}")

                # Check if container is in a layout
                if parent and parent.layout():
                    layout = parent.layout()
                    if hasattr(layout, "indexOf"):
                        index = layout.indexOf(container)
                        logger.info(f"  Layout index: {index}")

                    # For grid layouts, find the position
                    if isinstance(layout, QGridLayout):
                        found_position = None
                        for row in range(layout.rowCount()):
                            for col in range(layout.columnCount()):
                                item = layout.itemAtPosition(row, col)
                                if item and item.widget() == container:
                                    found_position = (row, col)
                                    break
                            if found_position:
                                break
                        logger.info(f"  Grid position: {found_position}")

            # Also debug the layout itself
            if hasattr(self.ui, "page_start") and hasattr(
                self.ui.page_start, "_container_layout_setup"
            ):
                layout = self.ui.page_start._container_layout_setup
                logger.info(f"=== Grid Layout Debug ===")
                logger.info(f"Grid rows: {layout.rowCount()}")
                logger.info(f"Grid columns: {layout.columnCount()}")
                logger.info(f"Grid item count: {layout.count()}")

                # Check each position in the grid
                widget_count_in_grid = 0
                for row in range(layout.rowCount()):
                    for col in range(layout.columnCount()):
                        item = layout.itemAtPosition(row, col)
                        if item and item.widget():
                            widget_count_in_grid += 1
                            widget = item.widget()
                            widget_id = getattr(widget, "container_id", "unknown")
                            logger.info(f"  Position ({row}, {col}): {widget_id}")

                logger.info(f"Total widgets in grid: {widget_count_in_grid}")

        except Exception as e:
            logger.error(f"Error in debug_containers: {e}")

    def reset_layout_references(self, target_widget):
        """Reset layout references when clearing containers"""
        try:
            if target_widget is None:
                return

            if hasattr(target_widget, "_container_layout_setup"):
                delattr(target_widget, "_container_layout_setup")
            if hasattr(target_widget, "_container_scroll_area"):
                # Remove the scroll area widget if it exists
                scroll_area = target_widget._container_scroll_area
                if scroll_area and scroll_area.parent():
                    scroll_area.parent().layout().removeWidget(scroll_area)
                    scroll_area.deleteLater()
                delattr(target_widget, "_container_scroll_area")
            if hasattr(target_widget, "_container_widget"):
                delattr(target_widget, "_container_widget")

            # Also clear the target widget's layout if it exists
            existing_layout = target_widget.layout()
            if existing_layout:
                # Remove all items from the layout
                while existing_layout.count():
                    item = existing_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                # Delete the layout itself
                existing_layout.deleteLater()
                target_widget.setLayout(None)

            logger.info(f"Reset layout references for {target_widget.objectName()}")
        except Exception as e:
            logger.error(f"Error resetting layout references: {e}")
