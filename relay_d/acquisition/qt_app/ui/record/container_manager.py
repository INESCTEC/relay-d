import time
import traceback

from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer

from relay_d.utils.coloring_logger import logger


def pump_events(ms: int):
    """Wait ``ms`` milliseconds while still processing Qt events (the window stays responsive).

    Falls back to a plain sleep when there is no running QCoreApplication.
    """
    if QCoreApplication.instance() is None:
        time.sleep(ms / 1000.0)
        return
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


class ContainerManager:
    """Owns the topic-container lifecycle for the record page.

    Builds containers from the YAML config, keeps their status text in sync
    with the subscribers, and handles layout refresh/resize. ``RecordPage``
    delegates to this class and keeps only the recording and config wiring.
    """

    def __init__(
        self,
        ui_instance,
        yaml_parser,
        topic_containers,
        topic_subscribers,
        container_visualizer,
    ):
        self.ui = ui_instance
        self.yaml_parser = yaml_parser
        self.topic_containers = topic_containers
        self.topic_subscribers = topic_subscribers
        self.container_visualizer = container_visualizer
        self._creating = False  # guards against re-entry while events are pumped

    def create_containers(self):
        """Create containers based on YAML configuration data - WITH SUBSCRIBERS AND VISUALIZATION"""
        if self._creating:
            logger.warning("Container creation already in progress; ignoring the request")
            return False
        self._creating = True
        try:
            # Clear existing containers, subscribers, and visualizations first
            self.container_visualizer.stop_all_visualizations()
            self.topic_containers.clear_all_containers()
            self.topic_subscribers.clear_all_subscribers()

            # Small delay to ensure cleanup is complete (the UI keeps processing events)
            pump_events(100)

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

            logger.info(
                f"Successfully created {len(created_containers)} containers (subscribers will start on record)"
            )
            return True

        except Exception as e:
            logger.error(f"Exception occurred in create_containers: {e}")
            traceback.print_exc()
            return False
        finally:
            self._creating = False

    def set_statuses_by_data(self, with_data, without_data):
        """Set every container's status text depending on whether it has data."""
        for container in self.topic_containers.containers:
            container_id = getattr(container, "container_id", None)
            if container_id and self.topic_subscribers.has_data(container_id):
                status = with_data
            else:
                status = without_data
            self.topic_containers.update_container_info(container, {"status": status})

    def set_status_for_all(self, status):
        """Set the same status text on every container that has an id."""
        for container in self.topic_containers.containers:
            if getattr(container, "container_id", None):
                self.topic_containers.update_container_info(
                    container, {"status": status}
                )

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
